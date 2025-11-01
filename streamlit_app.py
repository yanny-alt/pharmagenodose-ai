"""
🧬 PharmaGenoDose Streamlit Application
STUNNING UI + REAL MODEL PREDICTIONS
Conference-Ready Interactive Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os


sys.path.insert(0, os.path.dirname(__file__))

try:
    from pharmagenodose import PharmaGenoDoseFramework
except ImportError as e:
    st.error(f"❌ Failed to import PharmaGenoDoseFramework: {e}")
    st.stop()





# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🧬 PharmaGenoDose",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== PROFESSIONAL HEALTHCARE CSS ====================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles - CLEAN WHITE BACKGROUND */
    .stApp {
        background: #ffffff;
        font-family: 'Inter', sans-serif;
        color: #2d3748;
    }
    
    /* Main Content Area - Clean white cards */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        background: transparent;
    }
    
    /* Header Styling - Professional teal gradient */
    .main-header {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        font-size: 1.2rem;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Metric Cards - Clean teal theme */
    .metric-card {
        background: linear-gradient(135deg, #f0fdfa 0%, #ccfbf1 100%);
        padding: 20px;
        border-radius: 12px;
        color: #0f766e;
        text-align: center;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.15);
        border: 1px solid #99f6e4;
        margin: 8px 0;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(13, 148, 136, 0.2);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 8px 0;
        color: #0d9488;
    }
    
    .metric-label {
        font-size: 0.85rem;
        opacity: 0.8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
    }
    
    /* Buttons - Professional teal */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        padding: 12px 24px;
        border: none;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(13, 148, 136, 0.4);
        background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
    }
    
    /* Tab Styling - Clean and modern */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #f8fafc;
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        padding: 12px 20px;
        font-weight: 500;
        color: #64748b;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f0fdfa;
        color: #0d9488;
        border-color: #0d9488;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%) !important;
        color: white !important;
        border-color: #0d9488 !important;
        box-shadow: 0 2px 8px rgba(13, 148, 136, 0.3);
    }
    
    /* Input Fields */
        .stSelectbox, .stNumberInput, .stSlider, .stMultiSelect, .stRadio {
        background: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }

    /* Make form labels stand out - TEAL COLOR */
    label {
        color: #0d9488 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* Make metric numbers in Performance tab stand out */
    .stMetric {
        color: #0d9488 !important;
    }

    .stMetric label {
        color: #64748b !important;
        font-weight: 500 !important;
    }

    /* Make all headers and labels more visible */
    h1, h2, h3, h4, h5, h6 {
        color: #0d9488 !important;
    }

    /* Make sidebar labels visible */
    .stSidebar label {
        color: #0d9488 !important;
        font-weight: 600 !important;
    }

    /* Make selectbox text more visible */
    .stSelectbox [data-baseweb="select"] {
        color: #2d3748 !important;
    }

    /* Make number input text more visible */
    .stNumberInput input {
        color: #2d3748 !important;
    }

    /* Make radio button labels more visible */
    .stRadio label {
        color: #0d9488 !important;
        font-weight: 600 !important;
    }

    /* Make checkbox labels more visible */
    .stCheckbox label {
        color: #0d9488 !important;
        font-weight: 600 !important;
    }

    /* Make multiselect labels more visible */
    .stMultiSelect label {
        color: #0d9488 !important;
        font-weight: 600 !important;
    }
    /* Fix Training Dataset metrics visibility */
    .stMetric [data-testid="stMetricValue"] {
        color: #0d9488 !important;
        font-weight: 700 !important;
        font-size: 1.5rem !important;
    }
    
    .stMetric [data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-weight: 600 !important;
    }

    /* Fix radio button options (male/female) visibility */
    .stRadio [data-testid="stMarkdownContainer"] p {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }

    /* Fix checkbox label (Current Smoker) visibility */
    .stCheckbox [data-testid="stMarkdownContainer"] p {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }

    /* Fix number input text (70.00, 165.00) - make it dark/black */
    .stNumberInput input {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    /* Fix slider values visibility */
    .stSlider [data-testid="stMarkdownContainer"] p {
        color: #2d3748 !important;
        font-weight: 500 !important;
    }

    /* Fix selectbox dropdown text visibility */
    .stSelectbox [data-baseweb="select"] div {
        color: #ffffff !important;
        font-weight: 500 !important;
}

    /* Fix all input placeholder text */
    input::placeholder {
        color: #718096 !important;
    }

    /* Fix all text in input fields */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        color: #ffffff !important;
        font-weight: 500 !important;
        background-color: #0f766e !important;
    }

    /* Make section headers more prominent */
    h3 {
        color: #0d9488 !important;
        font-weight: 700 !important;
        margin-bottom: 1rem !important;
    }
    
    
    /* Cards - Clean white with subtle shadows */
    .custom-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin: 12px 0;
        border-left: 4px solid #0d9488;
        border: 1px solid #f1f5f9;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        box-shadow: 0 4px 16px rgba(0,0,0,0.12);
        transform: translateY(-1px);
    }
    
    /* Risk Badges - Clear and distinct */
    .risk-critical {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3);
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ea580c 0%, #f97316 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(234, 88, 12, 0.3);
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(217, 119, 6, 0.3);
    }
    
    .risk-standard {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 8px rgba(5, 150, 105, 0.3);
    }
    
    /* Result Display - Professional teal */
    .prediction-box {
        background: linear-gradient(135deg, #0d9488 0%, #14b8a6 100%);
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        color: white;
        box-shadow: 0 8px 25px rgba(13, 148, 136, 0.3);
        margin: 16px 0;
    }
    
    .prediction-value {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 16px 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-label {
        font-size: 1.1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    /* Alert Boxes - Clean and professional */
    .alert-box {
        padding: 16px;
        border-radius: 8px;
        margin: 12px 0;
        font-weight: 500;
        border-left: 4px solid;
    }
    
    .alert-error {
        background: #fef2f2;
        border-color: #dc2626;
        color: #991b1b;
    }
    
    .alert-warning {
        background: #fffbeb;
        border-color: #d97706;
        color: #92400e;
    }
    
    .alert-info {
        background: #f0f9ff;
        border-color: #0ea5e9;
        color: #075985;
    }
    
    .alert-success {
        background: #f0fdf4;
        border-color: #16a34a;
        color: #166534;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #f8fafc;
        border-radius: 8px;
        font-weight: 600;
        color: #0d9488;
        border: 1px solid #e2e8f0;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar for better appearance */
    ::-webkit-scrollbar {
        width: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    """Load the trained PharmaGenoDose model with beautiful loading animation"""
    try:
        model_path = "models/production_model.joblib"
        
        with st.spinner('🧬 Loading PharmaGenoDose AI Model...'):
            model_data = joblib.load(model_path)
        
        st.success("✅ Model loaded successfully! Ready to predict.")
        return model_data
        
    except FileNotFoundError:
        st.error(f"""
        ❌ **Model file not found!** 
        
        Please run your training code first to generate `pharmgenodose_model.pkl`
        
        **Steps:**
        1. Run: `python pharmgenodose.py`
        2. Wait for training to complete
        3. Verify `pharmgenodose_model.pkl` exists
        4. Restart this Streamlit app
        """)
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

def create_patient_dataframe(patient_data, feature_names):
    """Create a properly formatted DataFrame for prediction"""
    # Initialize with zeros for all features
    df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Fill in demographic features
    if 'Age_numeric' in df.columns:
        df['Age_numeric'] = patient_data['age']
    if 'Age_squared' in df.columns:
        df['Age_squared'] = patient_data['age'] ** 2
    if 'Age_log' in df.columns:
        df['Age_log'] = np.log(patient_data['age'])
    if 'Is_elderly' in df.columns:
        df['Is_elderly'] = 1 if patient_data['age'] >= 65 else 0
    if 'Age_risk_factor' in df.columns:
        df['Age_risk_factor'] = 0.8 if patient_data['age'] >= 65 else 1.0
    
    # Anthropometric features
    if 'Weight (kg)' in df.columns:
        df['Weight (kg)'] = patient_data['weight']
    if 'Height (cm)' in df.columns:
        df['Height (cm)'] = patient_data['height']
    
    # Calculate BMI and BSA
    height_m = patient_data['height'] / 100
    bmi = patient_data['weight'] / (height_m ** 2)
    bsa = 0.007184 * (patient_data['weight'] ** 0.425) * (patient_data['height'] ** 0.725)
    
    if 'BMI' in df.columns:
        df['BMI'] = bmi
    if 'BSA' in df.columns:
        df['BSA'] = bsa
    if 'BMI_category' in df.columns:
        if bmi < 18.5:
            df['BMI_category'] = 0
        elif bmi < 25:
            df['BMI_category'] = 1
        elif bmi < 30:
            df['BMI_category'] = 2
        else:
            df['BMI_category'] = 3
    
    # Genetic features
    cyp2c9_activity_map = {
        '*1/*1': 1.00, '*1/*2': 0.85, '*1/*3': 0.70,
        '*2/*2': 0.60, '*2/*3': 0.45, '*3/*3': 0.30
    }
    vkorc1_sensitivity_map = {
        'G/G': 0, 'G/A': 1, 'A/G': 1, 'A/A': 2
    }
    
    cyp_activity = cyp2c9_activity_map.get(patient_data['cyp2c9'], 0.88)
    vkorc_sens = vkorc1_sensitivity_map.get(patient_data['vkorc1'], 1)
    
    if 'CYP2C9_activity' in df.columns:
        df['CYP2C9_activity'] = cyp_activity
    if 'VKORC1_sensitivity' in df.columns:
        df['VKORC1_sensitivity'] = vkorc_sens
    
    # Genetic risk score
    if 'Genetic_risk_score' in df.columns:
        cyp_risk = (1 - cyp_activity) * 5
        vkorc_risk = vkorc_sens * 2.5
        df['Genetic_risk_score'] = cyp_risk + vkorc_risk
    
    # Medications
    drug_score = 0
    high_risk_score = 0
    
    if patient_data.get('medications'):
        for med in patient_data['medications']:
            med_col = f"{med}_binary"
            if med_col in df.columns:
                df[med_col] = 1
            
            # High risk drugs
            if med in ['Amiodarone (Cordarone)', 'Carbamazepine (Tegretol)', 
                      'Phenytoin (Dilantin)', 'Rifampin or Rifampicin']:
                drug_score += 3
                high_risk_score += 3
            else:
                drug_score += 1
    
    if 'High_risk_drug_score' in df.columns:
        df['High_risk_drug_score'] = high_risk_score
    if 'Total_drug_interaction_score' in df.columns:
        df['Total_drug_interaction_score'] = drug_score
    if 'Moderate_risk_drug_score' in df.columns:
        df['Moderate_risk_drug_score'] = drug_score - high_risk_score
    
    # Ethnicity (one-hot encoding)
    ethnicity_map = {
        'White': 'Race_White',
        'Asian': 'Race_Asian',
        'Black or African American': 'Race_Black or African American',
        'Unknown': 'Race_Unknown'
    }
    
    ethnicity_col = ethnicity_map.get(patient_data['ethnicity'])
    if ethnicity_col and ethnicity_col in df.columns:
        df[ethnicity_col] = 1
    
    # Gender
    if 'Gender_male' in df.columns:
        df['Gender_male'] = 1 if patient_data.get('gender', 'male') == 'male' else 0
    
    # Smoking
    if 'Is_smoker' in df.columns:
        df['Is_smoker'] = 1 if patient_data.get('smoker', False) else 0
    if 'Smoker_CYP_interaction' in df.columns:
        df['Smoker_CYP_interaction'] = df['Is_smoker'] * cyp_activity
    
    # Interaction terms
    if 'Gene_Age_interaction' in df.columns:
        df['Gene_Age_interaction'] = df['Genetic_risk_score'] * df['Age_numeric'] / 100
    if 'CYP_VKORC_interaction' in df.columns:
        df['CYP_VKORC_interaction'] = (1 - cyp_activity) * vkorc_sens
    if 'Gene_BSA_interaction' in df.columns:
        df['Gene_BSA_interaction'] = df['Genetic_risk_score'] * bsa
    if 'Age_BSA_interaction' in df.columns:
        df['Age_BSA_interaction'] = df['Age_numeric'] * bsa / 100
    
    # Amiodarone interactions
    if 'Amiodarone (Cordarone)_binary' in df.columns:
        amio_present = df['Amiodarone (Cordarone)_binary'].iloc[0]
        if 'Amiodarone_Age_interaction' in df.columns:
            df['Amiodarone_Age_interaction'] = amio_present * df['Age_numeric']
        if 'Amiodarone_CYP_interaction' in df.columns:
            df['Amiodarone_CYP_interaction'] = amio_present * (1 - cyp_activity) * 10
    
    # Target INR
    if 'Target_INR_numeric' in df.columns:
        df['Target_INR_numeric'] = patient_data.get('target_inr', 2.5)
    if 'INR_high_target' in df.columns:
        df['INR_high_target'] = 1 if patient_data.get('target_inr', 2.5) > 3.0 else 0
    if 'INR_low_target' in df.columns:
        df['INR_low_target'] = 1 if patient_data.get('target_inr', 2.5) < 2.0 else 0
    
    return df

def predict_dose(model_data, patient_data):
    """Make prediction using the loaded model"""
    try:
        
        from pharmagenodose import PharmaGenoDoseFramework
            
        
        # Create patient dataframe
        feature_names = model_data['feature_names']
        patient_df = create_patient_dataframe(patient_data, feature_names)
        
        # Reconstruct framework
        framework = PharmaGenoDoseFramework()
        framework.base_models = model_data['base_models']
        framework.meta_model = model_data['meta_model']
        framework.scaler = model_data['scaler']
        framework.feature_names = model_data['feature_names']
        framework.is_fitted = True
        framework.training_metrics = model_data['training_metrics']
        framework.clinical_thresholds = model_data['clinical_thresholds']
        
        # Predict
        result = framework.predict_with_uncertainty(patient_df)
        
        # Risk assessment
        risk = framework.clinical_risk_assessment(
            predicted_dose=result['prediction'][0],
            patient_age=patient_data['age'],
            patient_weight=patient_data['weight'],
            genetic_risk_score=patient_df['Genetic_risk_score'].iloc[0] if 'Genetic_risk_score' in patient_df.columns else None,
            drug_interaction_score=patient_df['Total_drug_interaction_score'].iloc[0] if 'Total_drug_interaction_score' in patient_df.columns else None
        )
        
        return {
            'predicted_dose': result['prediction'][0],
            'lower_bound': result['lower_bound'][0],
            'upper_bound': result['upper_bound'][0],
            'daily_dose': result['daily_dose'][0],
            'uncertainty': result['uncertainty_category'][0],
            'risk_assessment': risk
        }
        
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# ==================== MAIN APP ====================

# Header
st.markdown('<h1 class="main-header">🧬 PharmaGenoDose</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ensemble Machine Learning for Pharmacogenomic-Guided Warfarin Dosing</p>', unsafe_allow_html=True)

# Load model
model_data = load_model()

if model_data is None:
    st.stop()

# Display top metrics in beautiful cards
st.markdown("### 📊 Model Performance at a Glance")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">R² Score</div>
        <div class="metric-value">{model_data['training_metrics']['ensemble']['R2']:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MAE (mg/week)</div>
        <div class="metric-value">{model_data['training_metrics']['ensemble']['MAE']:.1f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Clinical Accuracy</div>
        <div class="metric-value">{model_data['training_metrics']['ensemble']['Clinical_Accuracy_20']:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Patients</div>
        <div class="metric-value">{model_data['training_metrics']['n_train'] + model_data['training_metrics']['n_test']}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== TABS ====================
tab1, tab2, tab3, tab4 = st.tabs(["📋 **Patient Dosing**", "💊 **Drug Interactions**", "🧬 **Genetics**", "📊 **Performance**"])

# ==================== TAB 1: PATIENT INPUT ====================
with tab1:
    st.markdown("## 🏥 Enter Patient Information")
    st.markdown("Fill in the patient details below to get personalized warfarin dosing recommendations")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 👤 Demographics")
        with st.container():
            age = st.slider("**Age** (years)", 18, 100, 50, help="Patient's age in years")
            weight = st.number_input("**Weight** (kg)", 30.0, 200.0, 70.0, 0.1, help="Patient's body weight")
            height = st.number_input("**Height** (cm)", 120.0, 220.0, 165.0, 0.1, help="Patient's height")
            
        st.markdown("### 🌍 Clinical Information")
        ethnicity = st.selectbox("**Ethnicity**", 
            ['White', 'Asian', 'Black or African American', 'Unknown'],
            help="Patient's self-reported ethnicity")
        
        col_gender, col_smoker = st.columns(2)
        with col_gender:
            gender = st.radio("**Gender**", ['male', 'female'])
        with col_smoker:
            smoker = st.checkbox("**Current Smoker**", help="Is the patient currently smoking?")
        
        target_inr = st.number_input("**Target INR**", 1.5, 4.0, 2.5, 0.1, 
                                     help="Target INR range (typically 2.0-3.0)")
        
    with col2:
        st.markdown("### 🧬 Genetic Profile")
        
        cyp2c9 = st.selectbox("**CYP2C9 Genotype**", 
            ['*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3'],
            help="CYP2C9 metabolizer genotype")
        
        # Show CYP2C9 activity indicator
        cyp_activities = {'*1/*1': 100, '*1/*2': 85, '*1/*3': 70, '*2/*2': 60, '*2/*3': 45, '*3/*3': 30}
        activity = cyp_activities[cyp2c9]
        
        if activity >= 90:
            color = "#51cf66"
            phenotype = "Normal Metabolizer"
        elif activity >= 60:
            color = "#ffd43b"
            phenotype = "Intermediate Metabolizer"
        else:
            color = "#ff6b6b"
            phenotype = "Poor Metabolizer"
        
        st.markdown(f"""
        <div style="background: {color}; padding: 15px; border-radius: 10px; color: white; font-weight: 600; text-align: center; margin-bottom: 20px;">
            {phenotype} - {activity}% Activity
        </div>
        """, unsafe_allow_html=True)
        
        vkorc1 = st.selectbox("**VKORC1 -1639 Genotype**",
            ['G/G', 'G/A', 'A/G', 'A/A'],
            help="VKORC1 sensitivity genotype")
        
        # Show VKORC1 sensitivity indicator
        vkorc_sens = {'G/G': 'Normal', 'G/A': 'Intermediate', 'A/G': 'Intermediate', 'A/A': 'High'}
        sens_level = vkorc_sens[vkorc1]
        
        sens_colors = {'Normal': '#51cf66', 'Intermediate': '#ffd43b', 'High': '#ff6b6b'}
        st.markdown(f"""
        <div style="background: {sens_colors[sens_level]}; padding: 15px; border-radius: 10px; color: white; font-weight: 600; text-align: center; margin-bottom: 20px;">
            {sens_level} Warfarin Sensitivity
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 💊 Concurrent Medications")
        medications = st.multiselect(
            "**Select all medications:**",
            [
                'Amiodarone (Cordarone)',
                'Carbamazepine (Tegretol)',
                'Phenytoin (Dilantin)',
                'Rifampin or Rifampicin',
                'Aspirin',
                'Simvastatin (Zocor)',
                'Atorvastatin (Lipitor)',
                'Acetaminophen or Paracetamol (Tylenol)'
            ],
            help="Select all medications the patient is currently taking"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Prediction button
    if st.button("🔮 **PREDICT WARFARIN DOSE**", type="primary"):
        patient_data = {
            'age': age,
            'weight': weight,
            'height': height,
            'ethnicity': ethnicity,
            'gender': gender,
            'smoker': smoker,
            'cyp2c9': cyp2c9,
            'vkorc1': vkorc1,
            'target_inr': target_inr,
            'medications': medications
        }
        
        with st.spinner('🧠 AI Model Computing Optimal Dose...'):
            result = predict_dose(model_data, patient_data)
        
        if result:
            st.balloons()
            
            # Main prediction display
            st.markdown(f"""
            <div class="prediction-box">
                <div class="prediction-label">Recommended Weekly Dose</div>
                <div class="prediction-value">{result['predicted_dose']:.1f} mg</div>
                <div class="prediction-label">({result['daily_dose']:.1f} mg per day)</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Three column display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="custom-card">
                    <h3 style="color: #667eea; margin-bottom: 15px;">📈 Confidence Interval</h3>
                    <p style="font-size: 1.8rem; font-weight: 700; color: #333; margin: 10px 0;">
                        {result['lower_bound']:.1f} - {result['upper_bound']:.1f}
                    </p>
                    <p style="color: #666; font-size: 0.9rem;">mg/week (95% CI)</p>
                    <p style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px; color: #667eea; font-weight: 600;">
                        Uncertainty: {result['uncertainty']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                risk = result['risk_assessment']
                risk_classes = {
                    'CRITICAL': 'risk-critical',
                    'HIGH': 'risk-high',
                    'MODERATE': 'risk-moderate',
                    'STANDARD': 'risk-standard'
                }
                risk_emojis = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MODERATE': '🟡',
                    'STANDARD': '🟢'
                }
                
                st.markdown(f"""
                <div class="custom-card">
                    <h3 style="color: #667eea; margin-bottom: 15px;">⚠️ Risk Assessment</h3>
                    <div style="text-align: center; margin: 20px 0;">
                        <span class="{risk_classes[risk['risk_category']]}">
                            {risk_emojis[risk['risk_category']]} {risk['risk_category']}
                        </span>
                    </div>
                    <p style="color: #666; font-size: 0.9rem; margin-top: 15px;">
                        Risk Score: {risk['risk_score']}/10
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="custom-card">
                    <h3 style="color: #667eea; margin-bottom: 15px;">📅 Daily Dosing</h3>
                    <p style="font-size: 1.8rem; font-weight: 700; color: #333; margin: 10px 0;">
                        {result['daily_dose']:.1f} mg
                    </p>
                    <p style="color: #666; font-size: 0.9rem;">per day</p>
                    <p style="margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px; color: #667eea; font-weight: 600;">
                        Weekly: {result['predicted_dose']:.1f} mg
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Clinical recommendations
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("## 📋 Clinical Recommendations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="alert-box alert-info">
                    <strong>💊 DOSING RECOMMENDATION</strong><br><br>
                    {risk['recommendation']}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="alert-box alert-warning">
                    <strong>🔬 MONITORING PROTOCOL</strong><br><br>
                    {risk['monitoring_schedule']}
                </div>
                """, unsafe_allow_html=True)
            
            # Alerts
            if risk['alerts']:
                st.markdown("<br>", unsafe_allow_html=True)
                alerts_html = "<br>".join([f"• {alert}" for alert in risk['alerts']])
                st.markdown(f"""
                <div class="alert-box alert-error">
                    <strong>🚨 CLINICAL ALERTS</strong><br><br>
                    {alerts_html}
                </div>
                """, unsafe_allow_html=True)
            
            # Risk factors
            if risk['risk_factors']:
                with st.expander("📌 **Detailed Risk Factors**", expanded=False):
                    for factor in risk['risk_factors']:
                        st.markdown(f"• {factor}")
            
            # Clinical notes
            st.markdown(f"""
            <div class="custom-card" style="margin-top: 20px; background: #f8f9fa;">
                <strong>📝 Clinical Notes:</strong><br>
                {risk['clinical_notes']}
            </div>
            """, unsafe_allow_html=True)

# ==================== TAB 2: DRUG INTERACTIONS ====================
with tab2:
    st.markdown("## 💊 Drug Interaction Reference")
    st.markdown("Comprehensive warfarin drug interaction database with clinical recommendations")
    
    drug_info = {
        'Amiodarone (Cordarone)': {
            'risk': 'HIGH',
            'mechanism': 'CYP2C9 inhibition - Significantly reduces warfarin metabolism',
            'dose_adjustment': '⬇️ Reduce warfarin dose by 30-50%',
            'monitoring': 'Check INR every 2-3 days initially, then weekly for 4 weeks',
            'clinical_notes': 'One of the most significant warfarin interactions. Start dose reduction immediately.'
        },
        'Rifampin or Rifampicin': {
            'risk': 'HIGH',
            'mechanism': 'CYP2C9 induction - Dramatically increases warfarin metabolism',
            'dose_adjustment': '⬆️ May need to increase dose by 30-50%',
            'monitoring': 'Check INR every 2-3 days. Monitor for therapeutic failure.',
            'clinical_notes': 'Effect begins within days. Monitor closely when starting or stopping rifampin.'
        },
        'Carbamazepine (Tegretol)': {
            'risk': 'HIGH',
            'mechanism': 'CYP450 induction - Increases hepatic metabolism',
            'dose_adjustment': '⬆️ Increase warfarin dose as needed based on INR',
            'monitoring': 'Frequent INR monitoring, especially in first 2 weeks',
            'clinical_notes': 'Enzyme induction effect may take 1-2 weeks to fully develop.'
        },
        'Phenytoin (Dilantin)': {
            'risk': 'HIGH',
            'mechanism': 'Complex bidirectional interaction - Both increases and decreases warfarin effect',
            'dose_adjustment': '⚠️ Unpredictable - Requires careful individual titration',
            'monitoring': 'Very frequent INR monitoring (every 2-3 days initially)',
            'clinical_notes': 'One of the most complex interactions. Avoid if possible.'
        },
        'Simvastatin (Zocor)': {
            'risk': 'MODERATE',
            'mechanism': 'CYP3A4 competition - Minor effect on warfarin metabolism',
            'dose_adjustment': '⬇️ May need 10-15% dose reduction',
            'monitoring': 'Check INR weekly for first month',
            'clinical_notes': 'Usually well-tolerated. Monitor for muscle pain (rhabdomyolysis risk).'
        },
        'Atorvastatin (Lipitor)': {
            'risk': 'MODERATE',
            'mechanism': 'Similar to simvastatin - CYP3A4 interaction',
            'dose_adjustment': '⬇️ Minimal adjustment usually needed',
            'monitoring': 'Standard INR monitoring',
            'clinical_notes': 'Lower interaction risk than simvastatin.'
        },
        'Aspirin': {
            'risk': 'MODERATE',
            'mechanism': 'Platelet inhibition - Additive anticoagulant effect (not metabolic)',
            'dose_adjustment': '⬇️ Consider 10-20% warfarin reduction. Monitor for bleeding.',
            'monitoring': 'Weekly INR. Assess for bleeding signs at each visit.',
            'clinical_notes': 'Common combination in cardiovascular disease. High bleeding risk.'
        },
        'Acetaminophen or Paracetamol (Tylenol)': {
            'risk': 'LOW',
            'mechanism': 'Minimal interaction at therapeutic doses (<2g/day)',
            'dose_adjustment': '➡️ Usually no adjustment needed for occasional use',
            'monitoring': 'Standard INR monitoring sufficient',
            'clinical_notes': 'High doses (>2g/day) or prolonged use may increase INR.'
        }
    }
    
    # Create interactive cards
    for drug, info in drug_info.items():
        risk_colors = {
            'HIGH': 'linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%)',
            'MODERATE': 'linear-gradient(135deg, #ff922b 0%, #fd7e14 100%)',
            'LOW': 'linear-gradient(135deg, #51cf66 0%, #2f9e44 100%)'
        }
        
        with st.expander(f"**{drug}** - {info['risk']} RISK", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div style="padding: 15px; background: #f8f9fa; border-radius: 10px; margin-bottom: 15px;">
                    <strong style="color: #667eea;">🔬 Mechanism of Action:</strong><br>
                    {info['mechanism']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="padding: 15px; background: #fff3bf; border-radius: 10px; margin-bottom: 15px;">
                    <strong style="color: #996800;">💊 Dose Adjustment:</strong><br>
                    {info['dose_adjustment']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="padding: 15px; background: #d0ebff; border-radius: 10px; margin-bottom: 15px;">
                    <strong style="color: #1864ab;">🔬 Monitoring Protocol:</strong><br>
                    {info['monitoring']}
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background: {risk_colors[info['risk']]}; padding: 20px; border-radius: 15px; color: white; text-align: center; margin-bottom: 15px;">
                    <div style="font-size: 3rem; margin-bottom: 10px;">
                        {'🔴' if info['risk'] == 'HIGH' else '🟠' if info['risk'] == 'MODERATE' else '🟢'}
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 700;">
                        {info['risk']}
                    </div>
                    <div style="font-size: 0.9rem; margin-top: 5px; opacity: 0.9;">
                        RISK LEVEL
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="padding: 15px; background: #e9ecef; border-radius: 10px; border-left: 5px solid #667eea;">
                <strong>📝 Clinical Notes:</strong><br>
                {info['clinical_notes']}
            </div>
            """, unsafe_allow_html=True)

# ==================== TAB 3: GENETICS ====================
with tab3:
    st.markdown("## 🧬 Pharmacogenetic Information")
    st.markdown("Understanding how genetics affect warfarin dosing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### CYP2C9 Variants")
        st.markdown("""
        **CYP2C9** is the primary enzyme responsible for metabolizing warfarin. 
        Genetic variants reduce enzyme activity, leading to slower drug clearance.
        """)
        
        cyp_data = pd.DataFrame({
            'Genotype': ['*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3'],
            'Activity': ['100%', '85%', '70%', '60%', '45%', '30%'],
            'Phenotype': ['Normal', 'Intermediate', 'Intermediate', 'Poor', 'Poor', 'Poor'],
            'Dose Impact': ['Standard', '↓ 10-15%', '↓ 25-30%', '↓ 35-40%', '↓ 50-55%', '↓ 65-70%']
        })
        
        st.dataframe(cyp_data, use_container_width=True, hide_index=True)
        
        st.info("""
        **Clinical Implication:** Patients with reduced CYP2C9 activity clear warfarin 
        more slowly, requiring lower doses to avoid over-anticoagulation and bleeding risk.
        """)
    
    with col2:
        st.markdown("### VKORC1 Variants")
        st.markdown("""
        **VKORC1** (Vitamin K Epoxide Reductase Complex 1) is warfarin's molecular target. 
        The -1639 G>A variant affects enzyme expression and warfarin sensitivity.
        """)
        
        vkorc_data = pd.DataFrame({
            'Genotype': ['G/G', 'G/A', 'A/A'],
            'Sensitivity': ['Normal', 'Intermediate', 'High'],
            'VKORC1 Expression': ['High', 'Medium', 'Low'],
            'Dose Impact': ['Standard', '↓ 20-25%', '↓ 40-50%']
        })
        
        st.dataframe(vkorc_data, use_container_width=True, hide_index=True)
        
        st.info("""
        **Clinical Implication:** The A allele reduces VKORC1 expression, making patients 
        more sensitive to warfarin. Lower doses are needed to achieve therapeutic INR.
        """)
    
    # Interactive genetics simulator
    st.markdown("---")
    st.markdown("## 🎮 Interactive Genetic Scenario Simulator")
    st.markdown("See how different genetic combinations affect warfarin dosing")
    
    sim_col1, sim_col2, sim_col3 = st.columns([1, 1, 1])
    
    with sim_col1:
        sim_cyp = st.selectbox("**CYP2C9 Genotype:**", 
                               ['*1/*1', '*1/*2', '*1/*3', '*2/*2', '*2/*3', '*3/*3'], 
                               key='sim_cyp')
    
    with sim_col2:
        sim_vkorc = st.selectbox("**VKORC1 Genotype:**", 
                                 ['G/G', 'G/A', 'A/A'], 
                                 key='sim_vkorc')
    
    with sim_col3:
        sim_age = st.slider("**Age:**", 18, 100, 50, key='sim_age')
    
    # Calculate estimated dose
    cyp_activity = {'*1/*1': 1.0, '*1/*2': 0.85, '*1/*3': 0.70, '*2/*2': 0.60, '*2/*3': 0.45, '*3/*3': 0.30}
    vkorc_sens = {'G/G': 0, 'G/A': 1, 'A/A': 2}
    
    base_dose = 42
    adjusted_dose = base_dose * cyp_activity[sim_cyp] - (vkorc_sens[sim_vkorc] * 8) - ((sim_age - 50) * 0.3)
    adjusted_dose = max(10, min(adjusted_dose, 80))
    
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">Estimated Weekly Dose</div>
        <div class="prediction-value">{adjusted_dose:.1f} mg</div>
        <div class="prediction-label">Based on genetics and age alone</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison table
    st.markdown("### 📊 Genetic Combination Examples")
    
    comparison_data = []
    for cyp in ['*1/*1', '*1/*3', '*3/*3']:
        for vkorc in ['G/G', 'G/A', 'A/A']:
            dose = 42 * cyp_activity[cyp] - (vkorc_sens[vkorc] * 8)
            comparison_data.append({
                'CYP2C9': cyp,
                'VKORC1': vkorc,
                'Est. Dose (mg/wk)': f"{dose:.1f}",
                'Phenotype': f"{['Normal', 'Intermediate', 'Poor'][['*1/*1', '*1/*3', '*3/*3'].index(cyp)]} / {['Normal', 'Intermediate', 'High'][['G/G', 'G/A', 'A/A'].index(vkorc)]}"
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# ==================== TAB 4: PERFORMANCE ====================
with tab4:
    st.markdown("## 📊 Model Performance & Validation")
    st.markdown("Comprehensive performance metrics from 4,671 patients in the IWPC dataset")
    
    # Overall metrics
    st.markdown("### 🎯 Overall Performance")
    metrics = model_data['training_metrics']['ensemble']
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">R² Score</div>
            <div class="metric-value">{metrics['R2']:.3f}</div>
            <div class="metric-label">Variance Explained</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">MAE</div>
            <div class="metric-value">{metrics['MAE']:.1f}</div>
            <div class="metric-label">mg/week</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">{metrics['RMSE']:.1f}</div>
            <div class="metric-label">mg/week</div>
        </div>
        """, unsafe_allow_html=True)
    
    with metric_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Clinical Accuracy</div>
            <div class="metric-value">{metrics['Clinical_Accuracy_20']:.1f}%</div>
            <div class="metric-label">Within ±20%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Ethnic performance
    st.markdown("### 👥 Ethnic Subgroup Performance")
    st.markdown("Ensuring equitable performance across diverse populations")
    
    ethnic_perf = model_data.get('ethnic_performance', {})
    
    if ethnic_perf:
        ethnic_data = []
        for ethnicity, data in ethnic_perf.items():
            ethnic_data.append({
                'Ethnic Group': ethnicity,
                'N Patients': data['n_patients'],
                'R²': f"{data['r2']:.3f}",
                'MAE (mg/wk)': f"{data['mae']:.1f}",
                'Clinical Accuracy (±20%)': f"{data['clinical_accuracy']:.1f}%",
                'Mean Dose': f"{data['mean_dose']:.1f} ± {data['std_dose']:.1f}"
            })
        
        ethnic_df = pd.DataFrame(ethnic_data)
        st.dataframe(ethnic_df, use_container_width=True, hide_index=True)
        
        st.info("""
        **Note:** Performance varies across ethnic groups due to differences in:
        - Genetic variant frequencies
        - Body composition
        - Dietary vitamin K intake
        - Sample size in training data
        """)
    else:
        st.warning("Ethnic performance data not available in this model version")
    
    # Model composition
    st.markdown("### 🤖 Ensemble Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Base Models:**
        - XGBoost (Gradient Boosting)
        - LightGBM (Fast Gradient Boosting)
        - Random Forest
        - Gradient Boosting Trees
        - ElasticNet (Linear Model)
        - Ridge Regression
        - Support Vector Regression
        - Multi-Layer Perceptron (Neural Network)
        """)
    
    with col2:
        st.markdown("""
        **Key Features:**
        - 45 engineered pharmacogenomic features
        - Genetic variant activity scores
        - Drug interaction terms
        - Age × BSA interactions
        - CYP × VKORC interactions
        - Uncertainty quantification
        - 4-tier risk stratification
        """)
    
    # Training info
    st.markdown("### 📚 Training Dataset")
    
    info_col1, info_col2, info_col3 = st.columns(3)
    
    with info_col1:
        st.metric("Training Set", f"{model_data['training_metrics']['n_train']} patients")
    
    with info_col2:
        st.metric("Test Set", f"{model_data['training_metrics']['n_test']} patients")
    
    with info_col3:
        total = model_data['training_metrics']['n_train'] + model_data['training_metrics']['n_test']
        st.metric("Total Dataset", f"{total} patients")
    
    # Key findings
    st.markdown("### 🔬 Key Research Findings")
    
    st.markdown("""
    <div class="custom-card">
        <h4 style="color: #667eea;">📈 Feature Importance Insights</h4>
        <ul>
            <li><strong>Age × BSA interaction:</strong> 13.6% importance - Most predictive single feature</li>
            <li><strong>Genetic interactions:</strong> CYP2C9 × VKORC1 more predictive than individual markers</li>
            <li><strong>Drug interactions:</strong> Amiodarone interactions show 10%+ importance</li>
            <li><strong>Individual genetic variants:</strong> Each <3% importance alone</li>
            <li><strong>Asian patients:</strong> Lower doses needed (R²=0.346, MAE=6.3 mg/wk)</li>
            <li><strong>Black/African American:</strong> Highest dose requirements (39.4±14.2 mg/wk) but lower R²=0.277</li>
            <li><strong>White patients:</strong> Moderate performance (R²=0.412, MAE=8.5 mg/wk)</li>
            <li><strong>Model disparity:</strong> Need for population-specific feature engineering</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; color: #666;">
    <strong>🧬 PharmaGenoDose</strong> | Ensemble Machine Learning for Warfarin Dosing<br>
    University of Nigeria, Faculty of Pharmaceutical Sciences<br>
    <em>Research by Favour Igwezeke</em>
</div>
""", unsafe_allow_html=True)