import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# 1. Page Configuration
st.set_page_config(
    page_title="Chronic Kidney Disease Screening App",
    page_icon="🩺",
    layout="wide"
)

# 2. Load Trained Pipeline (Preprocessor + Gradient Boosting Model)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "ckd_best_model.joblib"

@st.cache_resource
def load_model_pipeline():
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model pipeline: {e}")
        return None

pipeline = load_model_pipeline()
# 3. App Title & Description
st.title("🩺 Chronic Kidney Disease (CKD) Decision Support System")
st.markdown("""
This web application utilizes a trained **Gradient Tree Boosting** machine learning model to screen patients for Chronic Kidney Disease based on 51 clinical, biochemical and demographic features.
""")

if pipeline is None:
    st.warning("⚠️ Model pipeline `ckd_best_model.joblib` not found. Please run your Jupyter notebook up to Cell 72 to generate and save the model file.")
    st.stop()

# 4. Sidebar Configuration for Threshold Tuning (Controlling False Positives)
st.sidebar.header("⚙️ Deployment Settings")
st.sidebar.markdown("### False Positive Control")
threshold = st.sidebar.slider(
    "Decision Probability Threshold",
    min_value=0.50,
    max_value=0.95,
    value=0.80,
    step=0.05,
    help="Higher thresholds reduce false positives for healthy patients but may increase missed CKD cases."
)

st.sidebar.info(f"Current Threshold: **{threshold:.2f}**\n\n- Predictions $\ge$ {threshold} $\ rightarrow$ **CKD Positive**\n- Predictions $<$ {threshold} $\ srightarrow$ **Healthy / Low Risk**")

st.markdown("---")
st.subheader("📁 Upload Batch Patient CSV File")
uploaded_file = st.file_uploader("Upload a CSV file containing patient rows", type=["csv"])

if uploaded_file is not None:
    batch_df = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", batch_df.head(3))
    
    if st.button("🚀 Run Batch Prediction"):
        # Drop ID/target columns if present in uploaded csv
        clean_batch = batch_df.drop(columns=[col for col in ['PatientID', 'DoctorInCharge', 'Diagnosis'] if col in batch_df.columns], errors='ignore')
        
        # Predict probabilities
        batch_probs = pipeline.predict_proba(clean_batch)[:, 1]
        batch_preds = (batch_probs >= threshold).astype(int)
        
        # Display results table
        results_output = batch_df.copy()
        results_output['Predicted_Probability'] = batch_probs.round(4)
        results_output['Prediction'] = batch_preds
        results_output['Status'] = results_output['Prediction'].map({1: 'CKD Positive 🚨', 0: 'Healthy ✅'})
        
        st.success("Batch prediction complete!")
        st.dataframe(results_output[['Age', 'SerumCreatinine', 'GFR', 'Predicted_Probability', 'Status']])

st.markdown(

    "<h1 style='font-size:24px;'>OR</h1>",
    unsafe_allow_html=True
)


# 5. Patient Data Input Form
st.subheader("📋 Enter Patient Clinical & Laboratory Parameters")

with st.form("patient_form"):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input("Age (Years)", min_value=20, max_value=90, value=50)
        gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        bmi = st.number_input("BMI (kg/m²)", min_value=15.0, max_value=40.0, value=25.0)
        systolic_bp = st.number_input("Systolic BP (mmHg)", min_value=90, max_value=180, value=120)
        diastolic_bp = st.number_input("Diastolic BP (mmHg)", min_value=60, max_value=120, value=80)
        fasting_bs = st.number_input("Fasting Blood Sugar", min_value=70.0, max_value=200.0, value=100.0)
        hba1c = st.number_input("HbA1c (%)", min_value=4.0, max_value=10.0, value=5.5)

    with col2:
        serum_creatinine = st.number_input("Serum Creatinine (mg/dL)", min_value=0.5, max_value=5.0, value=1.2)
        bun_levels = st.number_input("BUN Levels (mg/dL)", min_value=5.0, max_value=50.0, value=15.0)
        gfr = st.number_input("GFR (mL/min/1.73m²)", min_value=15.0, max_value=120.0, value=90.0)
        protein_urine = st.number_input("Protein In Urine", min_value=0.0, max_value=5.0, value=0.1)
        acr = st.number_input("ACR (Albumin-Creatinine)", min_value=0.1, max_value=300.0, value=10.0)
        hemoglobin = st.number_input("Hemoglobin Levels", min_value=10.0, max_value=18.0, value=14.0)
        chol_total = st.number_input("Cholesterol Total", min_value=100.0, max_value=300.0, value=200.0)

    with col3:
        chol_ldl = st.number_input("Cholesterol LDL", min_value=50.0, max_value=200.0, value=100.0)
        chol_hdl = st.number_input("Cholesterol HDL", min_value=20.0, max_value=100.0, value=50.0)
        chol_trig = st.number_input("Triglycerides", min_value=50.0, max_value=400.0, value=150.0)
        sodium = st.number_input("Serum Sodium", min_value=120.0, max_value=150.0, value=140.0)
        potassium = st.number_input("Serum Potassium", min_value=3.0, max_value=6.0, value=4.5)
        calcium = st.number_input("Serum Calcium", min_value=7.0, max_value=11.0, value=9.5)
        phosphorus = st.number_input("Serum Phosphorus", min_value=2.0, max_value=6.0, value=3.5)

    with col4:
        # Categorical & Binary defaults matching the training set distributions
        ethnicity = st.selectbox("Ethnicity", options=[0, 1, 2, 3], format_func=lambda x: f"Group {x}")
        socioeconomic = st.selectbox("Socioeconomic Status", options=[0, 1, 2], format_func=lambda x: ["Low", "Medium", "High"][x])
        education = st.selectbox("Education Level", options=[0, 1, 2, 3], format_func=lambda x: ["None", "High School", "Bachelor", "Graduate"][x])
        smoking = st.selectbox("Smoking", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        fh_kidney = st.selectbox("Family History Kidney Disease", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        fh_htn = st.selectbox("Family History Hypertension", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")
        history_aki = st.selectbox("Previous Acute Kidney Injury", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes")

    # Expandable section for remaining secondary inputs to keep UI clean
    with st.expander("🔍 Additional Lifestyle & Symptom Parameters (Optional Adjustments)"):
        ac1, ac2, ac3 = st.columns(3)
        with ac1:
            alcohol = st.slider("Alcohol Consumption", 0.0, 20.0, 5.0)
            physical_activity = st.slider("Physical Activity Score", 0.0, 10.0, 5.0)
            diet_quality = st.slider("Diet Quality Score", 0.0, 10.0, 5.0)
            sleep_quality = st.slider("Sleep Quality Score", 4.0, 10.0, 7.0)
            fh_diabetes = st.selectbox("Family History Diabetes", [0, 1])
            uti = st.selectbox("Urinary Tract Infections", [0, 1])
        with ac2:
            ace_inhibitors = st.selectbox("ACE Inhibitors Use", [0, 1])
            diuretics = st.selectbox("Diuretics Use", [0, 1])
            nsaids = st.slider("NSAIDs Use", 0.0, 5.0, 0.0)
            statins = st.selectbox("Statins Use", [0, 1])
            antidiabetic = st.selectbox("Antidiabetic Medications", [0, 1])
            edema = st.selectbox("Edema (Swelling)", [0, 1])
        with ac3:
            fatigue = st.slider("Fatigue Levels", 0.0, 7.0, 2.0)
            nausea = st.slider("Nausea / Vomiting", 0.0, 7.0, 0.0)
            cramps = st.slider("Muscle Cramps", 0.0, 7.0, 1.0)
            itching = st.slider("Itching Levels", 0.0, 10.0, 2.0)
            qol = st.slider("Quality of Life Score", 0.0, 100.0, 70.0)
            heavy_metals = st.selectbox("Heavy Metals Exposure", [0, 1])
            chem_exposure = st.selectbox("Chemical Exposure", [0, 1])
            water_quality = st.selectbox("Water Quality Risk", [0, 1])
            checkups = st.slider("Medical Checkups Frequency", 0.0, 4.0, 2.0)
            adherence = st.slider("Medication Adherence", 0.0, 10.0, 8.0)
            literacy = st.slider("Health Literacy", 0.0, 10.0, 7.0)

    submitted = st.form_submit_button("🚀 Run Clinical Prediction")

if submitted:
    # Build DataFrame matching the exact 51 feature columns expected by the model
    input_data = pd.DataFrame({
        'Age': [age], 'Gender': [gender], 'Ethnicity': [ethnicity], 'SocioeconomicStatus': [socioeconomic],
        'EducationLevel': [education], 'BMI': [bmi], 'Smoking': [smoking], 'AlcoholConsumption': [alcohol],
        'PhysicalActivity': [physical_activity], 'DietQuality': [diet_quality], 'SleepQuality': [sleep_quality],
        'FamilyHistoryKidneyDisease': [fh_kidney], 'FamilyHistoryHypertension': [fh_htn], 'FamilyHistoryDiabetes': [fh_diabetes],
        'PreviousAcuteKidneyInjury': [history_aki], 'UrinaryTractInfections': [uti], 'SystolicBP': [systolic_bp],
        'DiastolicBP': [diastolic_bp], 'FastingBloodSugar': [fasting_bs], 'HbA1c': [hba1c], 'SerumCreatinine': [serum_creatinine],
        'BUNLevels': [bun_levels], 'GFR': [gfr], 'ProteinInUrine': [protein_urine], 'ACR': [acr],
        'SerumElectrolytesSodium': [sodium], 'SerumElectrolytesPotassium': [potassium], 'SerumElectrolytesCalcium': [calcium],
        'SerumElectrolytesPhosphorus': [phosphorus], 'HemoglobinLevels': [hemoglobin], 'CholesterolTotal': [chol_total],
        'CholesterolLDL': [chol_ldl], 'CholesterolHDL': [chol_hdl], 'CholesterolTriglycerides': [chol_trig],
        'ACEInhibitors': [ace_inhibitors], 'Diuretics': [diuretics], 'NSAIDsUse': [nsaids], 'Statins': [statins],
        'AntidiabeticMedications': [antidiabetic], 'Edema': [edema], 'FatigueLevels': [fatigue], 'NauseaVomiting': [nausea],
        'MuscleCramps': [cramps], 'Itching': [itching], 'QualityOfLifeScore': [qol], 'HeavyMetalsExposure': [heavy_metals],
        'OccupationalExposureChemicals': [chem_exposure], 'WaterQuality': [water_quality],
        'MedicalCheckupsFrequency': [checkups], 'MedicationAdherence': [adherence], 'HealthLiteracy': [literacy]
    })

    

    # Predict Probability
    prob_ckd = pipeline.predict_proba(input_data)[:, 1][0]
    prediction = int(prob_ckd >= threshold)

    st.markdown("---")
    st.subheader("📊 Screening Results & Risk Stratification")

    res_col1, res_col2 = st.columns([1, 2])

    with res_col1:
        st.metric(label="Estimated CKD Probability", value=f"{prob_ckd * 100:.1f}%")
        if prediction == 1:
            st.error("🚨 **Result: Positive for Chronic Kidney Disease (High Risk)**")
        else:
            st.success("✅ **Result: Negative / Healthy (Low Risk)**")

    with res_col2:
        st.markdown("### Clinical Interpretation & Decision Support")
        if prob_ckd >= 0.90:
            st.warning("⚠️ **High Confidence Positive**: Biomarker values strongly indicate reduced renal clearance and elevated filtration stress. Nephrology consultation recommended.")
        elif prob_ckd >= threshold:
            st.warning("⚡ **Moderate Risk (Above Threshold)**: Patient flagged positive under current threshold settings. Confirmatory blood panel advised to rule out false positives.")
        elif prob_ckd >= 0.50:
            st.info("ℹ️ **Borderline / Intermediate Zone**: Below your custom false-positive threshold, but elevated. Monitor eGFR and serum creatinine periodically.")
        else:
            st.info("🟢 **Low Risk**: Renal biomarkers fall within normal parameters. Maintain routine annual checkups.")

# ==============================================================================
# CLINICAL DECISION SUPPORT & THRESHOLD REASONING FOR DEPLOYMENT:
# ==============================================================================
# 1. Class Imbalance Mitigation: 
#    The training dataset exhibits severe class imbalance (approx. 91.86% CKD 
#    positive vs. 8.14% healthy controls). Standard classifiers using a default 
#    0.50 probability cutoff tend to over-predict the majority class, leading 
#    to an unacceptable False Positive Rate (approx. 74%) where healthy patients 
#    are incorrectly flagged.
#
# 2. Minimizing False Positives (Type I Errors):
#    In clinical screening workflows, sending healthy patients down unnecessary 
#    diagnostic pathways causes anxiety and resource strain. By raising the 
#    decision threshold (defaulting to 0.90 in this app), we require a much 
#    higher level of model confidence before declaring a patient 'CKD Positive'.
#
# 3. The Clinical Safety Trade-off:
#    While raising the threshold successfully suppresses False Positives and 
#    boosts Specificity, it creates an inverse trade-off with Recall (Sensitivity). 
#    The interactive threshold slider allows medical evaluators to balance 
#    this trade-off live based on whether they prioritize strict screening 
#    (lower threshold) or high-precision confirmation (higher threshold).
# ==============================================================================            